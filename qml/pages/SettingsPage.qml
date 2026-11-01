import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    objectName: "settingsPage"
    property var controller
    property var processingController
    property string projectId: ""
    property string projectName: ""
    property var pageState: ({})
    property var draft: ({})
    property var projectOverrides: ({})
    property bool dirty: false
    property string message: ""
    signal leaveRequested()

    readonly property var fields: [
        {key:"worker_count", title:"Processing workers", options:["Automatic (recommended)","1 worker","2 workers","3 workers","4 workers"], values:["automatic",1,2,3,4]},
        {key:"memory_usage", title:"Memory usage", options:["Automatic (recommended)","Conservative"], values:["automatic","conservative"]},
        {key:"blur_policy", title:"Blur handling", options:["Automatic","Review Borderline","Keep All"], values:["automatic","review_borderline","keep_all"]},
        {key:"face_quality_preset", title:"Face-quality sensitivity", options:["Relaxed","Balanced","Strict"], values:["relaxed","balanced","strict"]},
        {key:"keepers_per_group", title:"Preferred photographs per duplicate group", options:["1","2","3","4","5"], values:[1,2,3,4,5]},
        {key:"prefer_open_eyes", title:"Prefer open eyes", options:["On","Off"], values:[true,false]},
        {key:"prioritize_primary_face_quality", title:"Prioritize face quality", options:["On","Off"], values:[true,false]}
    ]

    function reload() {
        pageState = controller ? controller.pageData(projectId) : {values:{}, overrides:{}, globalValues:{}, defaults:{}, cpuCores:1, cpuSource:"logical processors"}
        draft = Object.assign({}, pageState.values)
        projectOverrides = Object.assign({}, pageState.overrides)
        dirty = false
        message = ""
    }
    function choose(key, value) {
        draft = Object.assign({}, draft, {[key]:value})
        if (projectId !== "") projectOverrides = Object.assign({}, projectOverrides, {[key]:value})
        dirty = true
    }
    function inherit(key) {
        let copy = Object.assign({}, projectOverrides)
        delete copy[key]
        projectOverrides = copy
        draft = Object.assign({}, draft, {[key]:pageState.globalValues[key]})
        dirty = true
    }
    function saveNow() {
        let changes = projectId !== "" ? projectOverrides : draft
        if (!controller || !controller.save(projectId, changes)) {
            message = "Could not save settings. Check the selected values."
            return
        }
        reload()
        message = "Saved. Some analysis results may need to be refreshed to reflect these settings."
    }
    function save() {
        let selected = draft.worker_count
        let previous = pageState.values.worker_count
        if ((selected !== previous || (projectId !== "" && Object.prototype.hasOwnProperty.call(projectOverrides, "worker_count") && !Object.prototype.hasOwnProperty.call(pageState.overrides, "worker_count")))
                && typeof selected === "number" && controller && controller.requiresCpuConfirmation(selected)) cpuWarning.open()
        else saveNow()
    }
    function requestLeave() { if (dirty) unsavedDialog.open(); else leaveRequested() }
    Component.onCompleted: reload()
    onProjectIdChanged: reload()

    Rectangle { anchors.fill: parent; color: "#F7F8FA" }
    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth
        ColumnLayout {
            width: Math.min(parent.width, 840)
            x: Math.max(0, (parent.width-width)/2)
            spacing: 18
            Item { Layout.preferredHeight: 20 }
            Button { text: "← Back"; onClicked: root.requestLeave(); activeFocusOnTab: true }
            Label { text: root.projectId ? "Project Settings" : "Global Settings"; font.pixelSize: 30; font.bold: true; color: "#17233B" }
            Label { visible: root.projectId !== ""; text: root.projectName; font.pixelSize: 18; color: "#17233B" }
            Label {
                text: root.projectId ? "Customize how Room36 processes this project. Changes here won't affect your global preferences."
                                     : "Default preferences used across all your projects."
                wrapMode: Text.WordWrap; Layout.fillWidth: true; color: "#566278"
            }
            Repeater {
                model: root.fields
                delegate: Rectangle {
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: content.implicitHeight + 28
                    radius: 10; color: "white"; border.color: "#DCE2EB"
                    ColumnLayout {
                        id: content
                        anchors.left: parent.left; anchors.right: parent.right
                        anchors.top: parent.top; anchors.margins: 14
                        spacing: 8
                        Label { text: modelData.title; font.bold: true; color: "#17233B" }
                        RowLayout {
                            Layout.fillWidth: true
                            ComboBox {
                                id: selection
                                objectName: "setting_" + modelData.key
                                Layout.preferredWidth: 290
                                model: modelData.options
                                currentIndex: Math.max(0, modelData.values.indexOf(root.draft[modelData.key]))
                                onActivated: root.choose(modelData.key, modelData.values[index])
                            }
                            Label {
                                visible: root.projectId !== ""
                                text: Object.prototype.hasOwnProperty.call(root.projectOverrides, modelData.key) ? "Overridden for this project" : "Inherited from Global Settings"
                                color: "#566278"
                            }
                            Item { Layout.fillWidth: true }
                            Button {
                                visible: root.projectId !== ""
                                text: "Use global default"
                                enabled: Object.prototype.hasOwnProperty.call(root.projectOverrides, modelData.key)
                                onClicked: root.inherit(modelData.key)
                            }
                        }
                        Label {
                            visible: modelData.key === "worker_count"
                            text: "Available " + root.pageState.cpuCores + " " + root.pageState.cpuSource + ". Selected: " + (root.draft.worker_count === "automatic" ? "Automatic" : root.draft.worker_count + " workers") + ". This is a concurrency limit, not a CPU usage percentage."
                            wrapMode: Text.WordWrap; Layout.fillWidth: true; color: "#566278"
                        }
                    }
                }
            }
            Label { text: root.message; visible: text.length > 0; wrapMode: Text.WordWrap; Layout.fillWidth: true; color: "#334E77" }
            Button {
                text: "Refresh analysis"
                visible: root.projectId !== "" && root.message.length > 0 && root.processingController !== null
                onClicked: root.processingController.retry()
            }
            RowLayout {
                Button { text: "Save"; enabled: root.dirty; onClicked: root.save() }
                Button { text: "Cancel"; onClicked: root.requestLeave() }
                Item { Layout.fillWidth: true }
                Button { text: "Reset to Defaults"; onClicked: resetDialog.open() }
            }
            Item { Layout.preferredHeight: 24 }
        }
    }
    Dialog {
        id: cpuWarning
        title: "High CPU usage"
        modal: true; anchors.centerIn: parent; width: 460
        footer: DialogButtonBox {
            Button { text: "Cancel"; DialogButtonBox.buttonRole: DialogButtonBox.RejectRole }
            Button { text: "Continue Anyway"; DialogButtonBox.buttonRole: DialogButtonBox.AcceptRole }
        }
        onAccepted: root.saveNow()
        contentItem: Label { text: "You're assigning a large portion of your CPU cores to image processing. This may make your computer less responsive during scans, especially when using other applications."; wrapMode: Text.WordWrap; width: 420 }
    }
    Dialog {
        id: resetDialog
        title: "Reset settings?"
        modal: true; anchors.centerIn: parent; width: 460
        standardButtons: Dialog.Cancel | Dialog.Ok
        onAccepted: { if (root.controller) root.controller.reset(root.projectId); root.reload() }
        contentItem: Label { text: root.projectId ? "Remove all overrides for this project?" : "Restore global application defaults?" }
    }
    Dialog {
        id: unsavedDialog
        title: "Unsaved changes"
        modal: true; anchors.centerIn: parent; width: 460
        contentItem: Label { text: "Save changes before leaving?" }
        footer: DialogButtonBox {
            Button { text: "Save"; DialogButtonBox.buttonRole: DialogButtonBox.AcceptRole; onClicked: { root.save(); if (!root.dirty) root.leaveRequested() } }
            Button { text: "Discard"; DialogButtonBox.buttonRole: DialogButtonBox.DestructiveRole; onClicked: root.leaveRequested() }
            Button { text: "Cancel"; DialogButtonBox.buttonRole: DialogButtonBox.RejectRole }
        }
    }
}


