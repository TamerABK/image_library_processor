pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "nearDuplicatesPage"
    property var duplicates: null
    property var inspector: null
    property string viewSize: "Medium"
    property string pendingNewKeeper: ""
    readonly property var group: duplicates ? duplicates.selectedGroup : ({})
    readonly property bool detailOpen: Boolean(group.groupId)
    onDetailOpenChanged: if (!detailOpen) {
        replaceKeeperDialog.close()
        pendingNewKeeper = ""
    }
    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Component.onDestruction: if (inspector) inspector.close()

    ColumnLayout {
        anchors.fill: parent
        visible: !root.detailOpen
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            Components.ToolbarDropdownButton {
                objectName: "duplicateFilter"
                label: root.duplicates ? root.duplicates.groupModel.groupFilter : "Needs review"
                options: ["Needs review", "Reviewed", "All"]
                onOptionSelected: function(value) {
                    if (root.duplicates) root.duplicates.groupModel.setFilter(value)
                }
                Accessible.name: "Filter duplicate groups: " + label
            }
            Components.ToolbarDropdownButton {
                objectName: "duplicateSort"
                label: root.duplicates ? root.duplicates.groupModel.sortOrder : "Largest groups"
                options: ["Largest groups", "Smallest groups"]
                onOptionSelected: function(value) {
                    if (root.duplicates) root.duplicates.groupModel.setSort(value)
                }
                Accessible.name: "Sort duplicate groups: " + label
            }
            Item { Layout.fillWidth: true }
        }
        Text {
            objectName: "duplicateStateText"
            Layout.fillWidth: true
            visible: !root.duplicates || root.duplicates.state !== "COMPLETE"
                || root.duplicates.groupModel.visibleCount === 0
                || root.duplicates.rankingState !== "COMPLETE"
            text: !root.duplicates ? "Finding near duplicates…"
                : root.duplicates.state === "FAILED"
                    ? "Near-duplicate analysis is unavailable. " + root.duplicates.stageError
                : root.duplicates.state !== "COMPLETE" ? "Finding near duplicates…"
                : root.duplicates.groupModel.totalCount === 0
                    ? "No near-duplicate groups were found."
                : root.duplicates.groupModel.visibleCount === 0
                    ? "No duplicate groups match this filter."
                : root.duplicates.rankingState === "FAILED" || root.duplicates.rankingState === "CANCELLED"
                    ? "Duplicate groups found. Keeper recommendations are unavailable."
                : root.duplicates.rankingState !== "COMPLETE"
                    ? "Duplicate groups found. Choosing recommended keepers…" : ""
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            font.pixelSize: typography.body
        }
        Components.PillButton {
            objectName: "duplicateRetry"
            variant: "accent"
            visible: root.duplicates !== null && root.duplicates.state === "FAILED"
            text: "Retry duplicate analysis"
            onClicked: if (root.duplicates) root.duplicates.retryProcessing()
            Accessible.name: "Retry project processing"
        }
        Components.DuplicateGroupGrid {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: root.duplicates !== null && root.duplicates.state === "COMPLETE"
            groupModel: root.duplicates ? root.duplicates.groupModel : null
            viewSize: root.viewSize
            onGroupOpened: function(groupId) { if (root.duplicates) root.duplicates.openGroup(groupId) }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        visible: root.detailOpen
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            Components.PillButton {
                objectName: "duplicateBack"
                variant: "translucent"
                text: "← Back"
                onClicked: if (root.duplicates) root.duplicates.back()
                Accessible.name: "Back to duplicate groups"
            }
            Text {
                Layout.fillWidth: true
                text: root.group.photoCount + " photos in this group"
                color: theme.mainBlue
                font.pixelSize: 22
            }
            Components.PillButton {
                objectName: "duplicateConfirm"
                variant: "accent"
                text: "Keep selected · Remove alternatives"
                enabled: root.duplicates !== null && !root.duplicates.busy
                    && root.group.recommendationState === "ready"
                onClicked: root.duplicates.confirmGroup()
                Accessible.name: "Confirm keepers and mark alternatives for future removal"
            }
        }
        Text {
            Layout.fillWidth: true
            visible: root.detailOpen && root.group.recommendationState !== "ready"
            text: root.group.recommendationState === "unavailable"
                ? "Keeper recommendation is unavailable. Retry project processing to try again."
                : "Keeper recommendation pending…"
            color: theme.mutedText
            font.pixelSize: typography.body
        }
        Text {
            objectName: "duplicateConflictMessage"
            Layout.fillWidth: true
            visible: root.detailOpen && Boolean(root.group.hasConflict)
            text: "Decision conflict: one or more photos have different choices in another review. Resolve those choices before confirming this group."
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            font.pixelSize: typography.body
            Accessible.name: text
        }
        Text {
            Layout.fillWidth: true
            visible: root.detailOpen && root.group.explanation && root.group.explanation.length > 0
            text: root.group.explanation ? "Recommended: " + root.group.explanation.join(" · ") : ""
            color: theme.mutedText
            wrapMode: Text.WordWrap
            font.pixelSize: typography.caption
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12
            Components.PhotoGrid {
                objectName: "duplicateDetailGrid"
                Layout.fillWidth: true
                Layout.fillHeight: true
                photoModel: root.duplicates ? root.duplicates.detailPhotoModel : null
                viewSize: root.viewSize
                openOnClick: true
                emptyText: "No photos are available for this group."
                onPhotoActivated: function(identity) {
                    if (root.inspector && root.duplicates)
                        root.inspector.openFromModel(root.duplicates.detailPhotoModel, identity)
                }
            }
            ListView {
                id: memberList
                objectName: "duplicateMemberList"
                Layout.preferredWidth: Math.min(300, Math.max(220, root.width * 0.30))
                Layout.fillHeight: true
                clip: true
                spacing: 8
                model: root.duplicates ? root.duplicates.detailMembers : []
                ScrollBar.vertical: ScrollBar {}
                delegate: Rectangle {
                    id: memberRow
                    required property var modelData
                    width: memberList.width - 14
                    height: 120
                    radius: 14
                    color: theme.panelFill
                    border.width: 1
                    border.color: theme.separator
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 4
                        Text {
                            Layout.fillWidth: true
                            text: memberRow.modelData.filename
                            color: theme.mainBlue
                            elide: Text.ElideMiddle
                            font.pixelSize: typography.body
                        }
                        Text {
                            text: memberRow.modelData.intent === "REMOVE" ? "Remove" :
                                memberRow.modelData.keeper ? "Current keeper" :
                                memberRow.modelData.recommended ? "Recommended" : "Alternative"
                            color: theme.mutedText
                            font.pixelSize: typography.caption
                        }
                        RowLayout {
                            Components.PillButton {
                                objectName: "duplicateSetKeeper"
                                variant: "translucent"
                                text: root.group.currentKeepers && root.group.currentKeepers.length > 1
                                    ? "Replace keeper…" : "Set keeper"
                                enabled: root.duplicates !== null && !root.duplicates.busy
                                    && root.group.recommendationState === "ready"
                                    && !memberRow.modelData.keeper
                                onClicked: {
                                    if (root.group.currentKeepers.length > 1) {
                                        root.pendingNewKeeper = memberRow.modelData.photoId
                                        replaceKeeperDialog.open()
                                    } else {
                                        root.duplicates.setKeeper(memberRow.modelData.photoId)
                                    }
                                }
                                Accessible.name: text + " " + memberRow.modelData.filename
                            }
                            Components.PillButton {
                                objectName: "duplicateToggleRemove"
                                variant: "translucent"
                                text: memberRow.modelData.intent === "REMOVE" ? "Restore" : "Remove"
                                enabled: root.duplicates !== null && !root.duplicates.busy
                                    && root.group.recommendationState === "ready"
                                    && (!memberRow.modelData.keeper || memberRow.modelData.intent === "REMOVE")
                                onClicked: root.duplicates.setIntent(memberRow.modelData.photoId,
                                    memberRow.modelData.intent === "REMOVE" ? "UNDECIDED" : "REMOVE")
                                Accessible.name: text + " " + memberRow.modelData.filename
                            }
                        }
                    }
                }
            }
        }
    }
    Dialog {
        id: replaceKeeperDialog
        objectName: "duplicateReplaceKeeperDialog"
        parent: Overlay.overlay
        anchors.centerIn: parent
        modal: true
        title: "Choose keeper to replace"
        width: Math.min(420, root.width - 40)
        standardButtons: Dialog.NoButton
        background: Rectangle {
            radius: 20
            color: theme.panelFill
            border.width: 1
            border.color: theme.separator
        }
        ColumnLayout {
            width: replaceKeeperDialog.availableWidth
            spacing: 10
            Text {
                Layout.fillWidth: true
                text: "Which current keeper should this photo replace?"
                color: theme.mainBlue
                wrapMode: Text.WordWrap
                font.pixelSize: typography.body
            }
            Repeater {
                model: root.group.currentKeepers || []
                delegate: Components.PillButton {
                    required property string modelData
                    objectName: "duplicateReplaceKeeperOption"
                    Layout.fillWidth: true
                    variant: "translucent"
                    text: modelData.substring(Math.max(modelData.lastIndexOf("/"), modelData.lastIndexOf("\\")) + 1)
                    onClicked: {
                        root.duplicates.replaceKeeper(modelData, root.pendingNewKeeper)
                        replaceKeeperDialog.close()
                    }
                    Accessible.name: "Replace current keeper " + text
                }
            }
            Components.PillButton {
                objectName: "duplicateCancelReplacement"
                Layout.alignment: Qt.AlignRight
                variant: "translucent"
                text: "Cancel"
                onClicked: replaceKeeperDialog.close()
                Accessible.name: "Cancel keeper replacement"
            }
        }
    }
    Text {
        objectName: "duplicateReviewMessage"
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        visible: root.duplicates !== null && (root.duplicates.busy || root.duplicates.error.length > 0)
        text: root.duplicates ? (root.duplicates.busy ? "Saving duplicate review…" : root.duplicates.error) : ""
        color: theme.mainBlue
        font.pixelSize: typography.body
        Accessible.name: text
    }
    Component {
        id: duplicateInspectorDetails
        ColumnLayout {
            Text { text: "Duplicate group photo"; color: theme.mainBlue; font.pixelSize: 20 }
            Text { text: "Choose keepers and removal intents in the group review.";
                color: theme.mutedText; wrapMode: Text.WordWrap; Layout.fillWidth: true }
        }
    }
    Components.PhotoInspector {
        anchors.fill: parent
        controller: root.inspector
        detailsComponent: duplicateInspectorDetails
        showReviewActions: false
        visible: root.detailOpen && controller !== null && controller.opened
    }
}
