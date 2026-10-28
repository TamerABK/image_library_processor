pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "knownPeoplePage"
    property var people: null
    property var inspector: null
    property var favorites: null
    property var exporter: null
    property string viewSize: "Medium"
    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    readonly property bool detailOpen: people !== null && Boolean(people.selectedKnown.key)
    function openRename() {
        if (!root.people || !root.detailOpen) return
        knownNameField.text = root.people.selectedKnown.name || ""
        knownRenameDialog.open()
        knownNameField.forceActiveFocus()
    }

    ColumnLayout {
        anchors.fill: parent
        visible: !root.detailOpen
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            TextField {
                id: knownSearch
                objectName: "knownPeopleSearch"
                Layout.preferredWidth: Math.min(320, root.width * 0.45)
                implicitHeight: 42
                placeholderText: "Search names"
                text: root.people ? root.people.knownModel.searchQuery : ""
                onTextEdited: if (root.people) root.people.knownModel.setSearch(text)
                Accessible.name: "Search known people by name"
                background: Rectangle {
                    radius: height / 2
                    color: theme.panelRaisedFill
                    border.width: knownSearch.activeFocus ? 2 : 1
                    border.color: knownSearch.activeFocus ? theme.focusRing : theme.separator
                }
            }
            Components.ToolbarDropdownButton {
                objectName: "knownPeopleSort"
                label: root.people ? root.people.knownModel.sortOrder : "Name A–Z"
                options: ["Name A–Z", "Most photos", "Most faces"]
                onOptionSelected: function(value) { if (root.people) root.people.knownModel.setSort(value) }
                Accessible.name: "Sort known people: " + label
            }
            Item { Layout.fillWidth: true }
            Components.PillButton {
                objectName: "knownExportAll"
                variant: "accent"
                text: "Export all people"
                enabled: root.exporter !== null && !root.exporter.busy && !root.exporter.preparing
                    && root.people !== null && root.people.state === "COMPLETE"
                    && root.people.knownCount > 0
                onClicked: root.exporter.prepareAllPeople()
                Accessible.name: "Export all known people into person folders"
            }
        }
        Text {
            objectName: "knownPeopleStateText"
            Layout.fillWidth: true
            visible: root.people === null || root.people.state !== "COMPLETE" || root.people.knownModel.visibleCount === 0
            text: !root.people || root.people.state !== "COMPLETE"
                ? (root.people && root.people.state === "FAILED"
                    ? "People analysis is unavailable. " + root.people.stageError
                    : "Processing people…")
                : root.people.knownCount === 0 ? "No known people were found in this project."
                : "No names match your search."
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            font.pixelSize: typography.body
        }
        Components.PillButton {
            objectName: "knownPeopleRetry"
            variant: "accent"
            visible: root.people !== null && root.people.state === "FAILED"
            text: "Retry people analysis"
            onClicked: if (root.people) root.people.retryProcessing()
            Accessible.name: "Retry project processing"
        }
        Components.PeopleGrid {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: root.people !== null && root.people.state === "COMPLETE"
            peopleModel: root.people ? root.people.knownModel : null
            viewSize: root.viewSize
            known: true
            onPersonOpened: function(key) { if (root.people) root.people.openKnown(key) }
        }
    }
    Components.PeopleDetailPhotos {
        anchors.fill: parent
        visible: root.detailOpen
        person: root.people ? root.people.selectedKnown : ({})
        title: person.name || "Known person"
        photoModel: root.people ? root.people.detailPhotoModel : null
        viewSize: root.viewSize
        inspector: root.inspector
        favoritesController: root.favorites
        exporter: root.exporter
        allowPersonExport: true
        onBackRequested: if (root.people) root.people.back()
        nameActions: RowLayout {
            spacing: 4
            Components.IconButton {
                objectName: "knownRenameIcon"
                source: theme.iconUrl("known_person_pen.svg")
                buttonSize: 32
                iconSize: 20
                enabled: root.people !== null && !root.people.busy
                onClicked: root.openRename()
                accessibleLabel: "Rename known person"
            }
            Components.IconButton {
                objectName: "knownTrashIcon"
                source: theme.iconUrl("known_person_trash.svg")
                buttonSize: 32
                iconSize: 18
                enabled: root.people !== null && !root.people.busy
                onClicked: knownDeleteDialog.open()
                accessibleLabel: "Delete known person"
                background: Rectangle {
                    radius: width / 2
                    color: parent.down ? theme.midBlue : theme.mainBlue
                    border.width: parent.activeFocus ? 1 : 0
                    border.color: theme.focusRing
                }
            }
        }
    }
    Text {
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        visible: root.detailOpen && root.people !== null
            && (Boolean(root.people.busy) || (root.people.error || "").length > 0)
        text: root.people ? (root.people.busy ? (root.people.busyMessage || "")
            : (root.people.error || "")) : ""
        color: theme.mainBlue
        font.pixelSize: typography.body
        Accessible.name: text
    }
    Dialog {
        id: knownRenameDialog
        objectName: "knownRenameDialog"
        parent: Overlay.overlay
        anchors.centerIn: parent
        modal: true
        title: "Rename known person"
        width: Math.min(420, root.width - 40)
        standardButtons: Dialog.NoButton
        background: Rectangle {
            radius: 20
            color: theme.panelFill
            border.width: 1
            border.color: theme.separator
        }
        ColumnLayout {
            width: knownRenameDialog.availableWidth
            spacing: 12
            TextField {
                id: knownNameField
                objectName: "knownPersonName"
                Layout.fillWidth: true
                maximumLength: 100
                placeholderText: "Person name"
                Accessible.name: "Known person name"
                onAccepted: knownSaveName.clicked()
                implicitHeight: 42
                background: Rectangle {
                    radius: height / 2
                    color: theme.panelRaisedFill
                    border.width: knownNameField.activeFocus ? 2 : 1
                    border.color: knownNameField.activeFocus ? theme.focusRing : theme.separator
                }
            }
            Text {
                Layout.fillWidth: true
                visible: root.people !== null && (root.people.error || "").length > 0
                text: root.people ? (root.people.error || "") : ""
                color: theme.mainBlue
                wrapMode: Text.WordWrap
            }
            RowLayout {
                Layout.fillWidth: true
                Item { Layout.fillWidth: true }
                Components.PillButton {
                    objectName: "knownCancelRename"
                    variant: "translucent"
                    text: "Cancel"
                    onClicked: knownRenameDialog.close()
                }
                Components.PillButton {
                    id: knownSaveName
                    objectName: "knownSaveName"
                    variant: "accent"
                    text: "Save name"
                    enabled: root.people !== null && !root.people.busy
                    onClicked: if (root.people && root.people.renameKnown(knownNameField.text))
                        knownRenameDialog.close()
                    Accessible.name: "Save known person name"
                }
            }
        }
    }
    Dialog {
        id: knownDeleteDialog
        objectName: "knownDeleteDialog"
        parent: Overlay.overlay
        anchors.centerIn: parent
        modal: true
        title: "Delete known person?"
        width: Math.min(420, root.width - 40)
        standardButtons: Dialog.NoButton
        background: Rectangle {
            radius: 20
            color: theme.panelFill
            border.width: 1
            border.color: theme.separator
        }
        ColumnLayout {
            width: knownDeleteDialog.availableWidth
            spacing: 12
            Text {
                Layout.fillWidth: true
                text: "This removes the person and their face embeddings. Photographs stay in the project."
                color: theme.mainBlue
                wrapMode: Text.WordWrap
            }
            RowLayout {
                Layout.alignment: Qt.AlignRight
                Components.PillButton {
                    objectName: "knownCancelDelete"
                    variant: "translucent"
                    text: "Cancel"
                    onClicked: knownDeleteDialog.close()
                }
                Components.PillButton {
                    objectName: "knownConfirmDelete"
                    variant: "accent"
                    text: "Delete person"
                    enabled: root.people !== null && !root.people.busy
                    onClicked: if (root.people && root.people.deleteKnown()) knownDeleteDialog.close()
                    Accessible.name: "Confirm delete known person and embeddings"
                }
            }
        }
    }
}
